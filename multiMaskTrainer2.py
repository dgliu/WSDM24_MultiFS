import torch
import argparse
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
from modules import multiMask2

parser = argparse.ArgumentParser(description="multifs trainer")
parser.add_argument("--dataset", type=str, help="specify dataset", default="AliExpress-1")
parser.add_argument("--model", type=str, help="specify model", default="deepfm")


# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("--l2", type=float, help="L2 regularization", default=3e-6)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, default="save/", help="model save directory")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", default=False, help="mlp batch normalization")
parser.add_argument("--cross", type=int, help="cross layer", default=3)

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=0, help="device info")

# mask information
parser.add_argument("--mask_weight_init", type=float, default=0.5, help="mask weight initial value")
parser.add_argument("--scaling", type=float, default=3, help="mask scaling value")
parser.add_argument("--final_temp", type=float, default=100, help="final temperature")
parser.add_argument("--init_thre", type=float, default=0.95, help="mask_s init threshold")
parser.add_argument("--search_epoch", type=int, default=5, help="search epochs")
parser.add_argument("--rewind_epoch", type=int, default=0, help="rewind epoch")
parser.add_argument("--lambda1", type=float, default=0.1, help="share loss rate")
parser.add_argument("--lambda2_s", type=float, default=2e-8, help="regularization rate")
parser.add_argument("--lambda2", type=float, default=1e-9, help="regularization rate")
parser.add_argument("--lambda3", type=float, default=1e-8, help="regularization rate")
args = parser.parse_args()

my_seed = 2022
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'


class Trainer(object):
    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.epochs = opt["search_epoch"]
        self.rewind_epoch = opt["rewind_epoch"]
        self.share_lambda1 = opt["lambda1"]
        self.reg_lambda2_s = opt["lambda2_s"]
        self.reg_lambda2 = opt["lambda2"]
        self.reg_lambda3 = opt["lambda3"]
        self.temp_increase = opt["final_temp"] ** (1. / (opt["search_epoch"] - 1))
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = multiMask2.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optim = multiMask2.getOptim(self.network, opt["optimizer"], self.lr, self.l2)
        self.logger = trainUtils.get_log(opt['model'])
        self.model_opt = opt["model_opt"]
        self.mask_weight_s = None
        self.mask_weight_i = None
        self.mask_weight_j = None
        self.init_weight = None


    def train_on_batch(self, label, data, domain, retrain=False):
        self.network.train()
        self.network.zero_grad()
        data, label, domain = data.to(self.device), label.to(self.device), domain.to(self.device)
        logit0, logit1, logit0s, logit1s = self.network(data)
        logloss0 = self.criterion(logit0, label)
        logloss1 = self.criterion(logit1, label)
        logloss0s = self.criterion(logit0s, label)
        logloss1s = self.criterion(logit1s, label)
        if not retrain:
            regloss1_s = self.reg_lambda2_s * self.network.reg1_s()
            regloss1_i = self.reg_lambda2 * torch.mean(self.network.reg1_i() * (1 - domain))
            regloss1_j = self.reg_lambda2 * torch.mean(self.network.reg1_j() * domain)
            regloss2_ij = self.reg_lambda3 * self.network.reg2_ij()
            share_logloss = (torch.mean(logloss0s) + torch.mean(logloss1s)) / 2
            domain_logloss = torch.mean(logloss0 * (1 - domain) + logloss1 * domain)
            loss = self.share_lambda1 * share_logloss + domain_logloss + regloss1_s + regloss1_i + regloss1_j + regloss2_ij
        else:
            share_logloss = (torch.mean(logloss0s) + torch.mean(logloss1s)) / 2
            domain_logloss = torch.mean(logloss0 * (1 - domain) + logloss1 * domain)
            loss = self.share_lambda1 * share_logloss + domain_logloss
        loss.backward()
        for optim in self.optim:
            optim.step()
        return loss.item()

    def eval_on_batch(self, data, domain):
        self.network.eval()
        with torch.no_grad():
            data, domain = data.to(self.device), domain.to(self.device)
            logit0, logit1, logit0s, logit1s = self.network(data)
            if domain[0] == 0:
                logit = logit0
            else:
                logit = logit1
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def search(self):
        self.logger.info("ticket:{t}".format(t=self.network.ticket))
        self.logger.info("-----------------Begin Search-----------------")
        ds = self.dataloader.get_train_data("train", batch_size=self.bs)
        for epoch_idx in range(int(self.epochs)):
            train_loss = .0
            step = 0
            if epoch_idx > 0:
                self.network.temp *= self.temp_increase
            if epoch_idx == self.rewind_epoch:
                self.network.checkpoint()
            for feature, label, domain in ds:
                loss = self.train_on_batch(label, feature, domain)
                train_loss += loss
                step += 1
                if step % 1000 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
                    self.logger.info(self.network.compute_remaining_weights())
            train_loss /= step
            self.logger.info("Temp:{temp:.6f}".format(temp=self.network.temp))
            self.logger.info("Thre:{thre:.6f}".format(thre=self.network.thre.item()))
            val_auc, val_loss = self.evaluate_val("validation")
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))

            self.network.compute_remaining_weights()
        test_auc0, test_loss0 = self.evaluate_test("test", "0")
        test_auc1, test_loss1 = self.evaluate_test("test", "1")
        self.logger.info(
            "Test AUC0: {test_auc:.6f}, Test Loss0: {test_loss:.6f}".format(test_auc=test_auc0, test_loss=test_loss0))
        self.logger.info(
            "Test AUC1: {test_auc:.6f}, Test Loss1: {test_loss:.6f}".format(test_auc=test_auc1, test_loss=test_loss1))
        self.mask_weight_s = self.network.mask_embedding.mask_weight_s
        self.mask_weight_i = self.network.mask_embedding.mask_weight_i
        self.mask_weight_j = self.network.mask_embedding.mask_weight_j
        mask_s = self.network.mask_embedding.mask_weight_s.detach().cpu().numpy()
        mask_i = self.network.mask_embedding.mask_weight_i.detach().cpu().numpy()
        mask_j = self.network.mask_embedding.mask_weight_j.detach().cpu().numpy()
        s = (mask_s > 0).astype(int)
        i = (mask_i > 0).astype(int)
        j = (mask_j > 0).astype(int)
        s_i = ((mask_s > 0) & (mask_i > 0)).astype(int)
        s_j = ((mask_s > 0) & (mask_j > 0)).astype(int)
        self.logger.info("share_remain:{} | spec_i_remain:{} | spec_j_remian:{}".format(np.sum(s), np.sum(i-s_i), np.sum(j-s_j)))
        self.logger.info("spec_i & spec_j:{}".format(np.sum((i-s_i) & (j-s_j))))

    def evaluate_val(self, on: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on + "0", batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        for feature, label, domain in self.dataloader.get_data(on + "1", batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def evaluate_test(self, on: str, dom: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on + dom, batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def train(self, epochs):
        self.network.ticket = True
        self.network.rewind_weights()
        cur_auc = 0.0
        early_stop = False
        self.optim = multiMask2.getOptim(self.network, "adam", self.lr, self.l2)[:1]
        rate_s, rate_i, rate_j = self.network.compute_remaining_weights()

        self.logger.info("-----------------Begin Train-----------------")
        self.logger.info("Ticket:{t}".format(t=self.network.ticket))
        self.logger.info(
            "Feature s remain:{rate_s:.6f} | Feature i remain:{rate_i:.6f} | Feature j remain:{rate_j:.6f}".format(
                rate_s=rate_s, rate_i=rate_i, rate_j=rate_j))
        ds = self.dataloader.get_train_data("train", batch_size=self.bs)
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label, domain in ds:
                loss = self.train_on_batch(label, feature, domain, retrain=True)
                train_loss += loss
                step += 1
                if step % 1000 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            val_auc, val_loss = self.evaluate_val("validation")
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))

            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                test_auc0, test_loss0 = self.evaluate_test("test", "0")
                test_auc1, test_loss1 = self.evaluate_test("test", "1")
                self.logger.info(
                    "Early stop at epoch {epoch:d} | Test AUC0: {test_auc:.6f}, Test Loss0:{test_loss:.6f}".format(
                        epoch=epoch_idx, test_auc=test_auc0, test_loss=test_loss0))
                self.logger.info(
                    "Early stop at epoch {epoch:d} | Test AUC1: {test_auc:.6f}, Test Loss1:{test_loss:.6f}".format(
                        epoch=epoch_idx, test_auc=test_auc1, test_loss=test_loss1))
                break

        if not early_stop:
            test_auc0, test_loss0 = self.evaluate_test("test", "0")
            test_auc1, test_loss1 = self.evaluate_test("test", "1")
            self.logger.info("Final Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}".format(test_auc=test_auc0,
                                                                                                 test_loss=test_loss0))
            self.logger.info("Final Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}".format(test_auc=test_auc1,
                                                                                                 test_loss=test_loss1))


def main():
    sys.path.extend(["./modules", "./dataloader", "./utils"])
    if args.dataset.lower() == "AliExpress-1":
        field_dim = trainUtils.get_stats("data/AliExpress-1/stats_2")
        data_dir = "data/AliExpress-1/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    elif args.dataset.lower() == "AliExpress-2":
        field_dim = trainUtils.get_stats("data/AliExpress-2/stats_2")
        data_dir = "data/AliExpress-2/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    else:
        print("dataset error")
    model_opt = {
        "latent_dim": args.dim, "feat_num": feature, "field_num": field,
        "mlp_dropout": args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims": args.mlp_dims, "cross": args.cross,
        "mask_weight_initial": args.mask_weight_init, "mask_scaling": args.scaling, "init_thre": args.init_thre
    }

    opt = {
        "model_opt": model_opt, "dataset": args.dataset, "model": args.model, "lr": args.lr, "l2": args.l2,
        "bsize": args.bsize, "optimizer": args.optim, "data_dir": data_dir, "save_dir": args.save_dir,
        "cuda": args.cuda, "search_epoch": args.search_epoch, "rewind_epoch": args.rewind_epoch,
        "final_temp": args.final_temp, "lambda1": args.lambda1, "lambda2_s": args.lambda2_s,
        "lambda2": args.lambda2, "lambda3": args.lambda3
    }
    print(opt)
    trainer = Trainer(opt)
    trainer.search()
    trainer.train(args.max_epoch)


if __name__ == "__main__":
    """
    python multiMaskTrainer2.py --dataset 'AliExpress-1' --model 'deepfm'   
    """
    main()
