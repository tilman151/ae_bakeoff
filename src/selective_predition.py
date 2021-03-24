import os

import matplotlib.pyplot as plt
from sklearn import metrics

import building
import downstream
import run
from downstream.results import AbstractResults
from reproduce import Checkpoints


def reproduce(retrain, recalc_downstream, replications, gpu):
    load_checkpoints = not retrain
    load_downstream_results = (not recalc_downstream) and load_checkpoints

    checkpoints = Checkpoints("emnist", load_from_disk=load_checkpoints)
    checkpoints = train_all(checkpoints, replications, gpu)

    anomaly_detection = CoveragesDownstream("emnist", load_from_disk=load_downstream_results)
    anomaly_detection = do_anomaly_detection(checkpoints, anomaly_detection)

    anomaly_detection.render()


def do_anomaly_detection(checkpoints, anomaly_detection):
    for model_type in checkpoints.keys():
        for checkpoint in checkpoints[model_type]:
            if model_type not in anomaly_detection:
                anomaly_detection.add_coverages_for(model_type, checkpoint)

    return anomaly_detection


def train_all(checkpoints, replications, gpu):
    for model_type in run.AUTOENCODERS:
        for _ in range(replications):
            if model_type not in checkpoints or len(checkpoints[model_type]) < replications:
                checkpoints[model_type] = []
                checkpoints[model_type].append(run.run(model_type,
                                                       dataset="emnist",
                                                       batch_size=128,
                                                       gpu=gpu,
                                                       anomaly=True))
                checkpoints.save()

    return checkpoints


class CoveragesDownstream(AbstractResults):
    def add_coverages_for(self, model_type, checkpoint_path):
        coverages, risks, auc = self._get_test_coverages(model_type, checkpoint_path)
        if model_type not in self:
            self[model_type] = {'coverages': [],
                                'risks': [],
                                'aucs': []}
        self[model_type]['coverages'].append(coverages.tolist())
        self[model_type]['risks'].append(risks.tolist())
        self[model_type]['aucs'].append(auc)
        self.save()

    def _get_test_coverages(self, model_type, checkpoint_path):
        data_module = building.build_datamodule(self.dataset, anomaly=True)
        anomaly_detector = downstream.AnomalyDetection.from_autoencoder_checkpoint(model_type, data_module, checkpoint_path)
        _, _, _, coverages, risks, _ = anomaly_detector.get_test_roc(data_module)
        auc = metrics.auc(coverages, risks)

        return coverages, risks, auc

    def render(self):
        print('Render anomaly detection results...')
        fig = plt.figure(figsize=(16, 9))
        self._plot_coverage_risks()
        fig.tight_layout()
        plt.savefig(self._get_output_path())
        plt.close()

    def _plot_coverage_risks(self):
        downstream.plot_perfect_risk_coverage(plt.gca())
        for model_type in self.keys():
            downstream.plot_risk_coverage(plt.gca(),
                                          self[model_type]['coverages'],
                                          self[model_type]['risks'],
                                          self[model_type]['aucs'],
                                          title=model_type)
        plt.legend(loc='lower right')

    def _get_output_path(self):
        log_path = self._get_log_path()
        output_path = os.path.join(log_path, 'risk_coverage.png')

        return output_path

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'coverage_results.json')

        return results_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Replicate experiments for selective prediction.")
    parser.add_argument('--retrain', action='store_true', help='Retrain even if checkpoints are available')
    parser.add_argument('--recalc_downstream', action='store_true', help='Recalculate downstream tasks')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument("-r", "--replications", type=int, help="how many replications per autoencoder")
    parser.add_argument("--gpu", action="store_true", help="use GPU for training")
    opt = parser.parse_args()

    reproduce(opt.retrain, opt.recalc_downstream, opt.replications, opt.gpu)
