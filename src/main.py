#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 15-Jan-2024
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the main function.
# ---------------------------------------------------------------------------

import torch
import click
import logging

from dataset import CSP_Dataset, AECSP_Dataset
from pathlib import Path
from utils import Config
from optim import ForecastTrainer, AETrainer

@click.command()
@click.option('--exp_config', type = click.Path(exists=True), default=None, help='configuration file')
def main(exp_config):
    """Main function

    Args:
        exp_config (Path): configuration file path.
    """

    # Load configuration file
    config = Config(locals().copy())
    config.load_config(Path(exp_config))

    # Set up logger
    config.settings['log_path'] = Path(config.settings['log_path']).joinpath(config.settings['experiment'])

    if not Path.exists(Path(config.settings['log_path'])):
        Path.mkdir(Path(config.settings['log_path']), parents=True)

    if config.settings['train_forecast']:
        log_path = config.settings['log_path'].joinpath('log.txt')
    else:
        log_path = config.settings['log_path'].joinpath('log_test.txt')

    logging.basicConfig(level = logging.INFO,
                        filemode = 'w',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename = log_path)
    logger = logging.getLogger()

    logger.info('Log file is %s.' % (log_path))

    # create output path
    config.settings['output_path'] = Path(config.settings['output_path']).joinpath(config.settings['experiment'])
    if not Path.exists(Path(config.settings['output_path'])):
        Path.mkdir(Path(config.settings['output_path']), parents=True)

    # device
    if torch.cuda.is_available():
        config.settings["device"] = 'cuda'
    else:
        config.settings["device"] = 'cpu'
    logger.info('Device is %s.' % config.settings["device"])

    # ae model train + test
    if config.settings['pretrained_ae'] is None and config.settings['train_forecast']:

        pretrain_dataset = AECSP_Dataset(config)
        aetrainer = AETrainer(config)
        model_path = aetrainer.train(pretrain_dataset)
        aetrainer.test(pretrain_dataset)

        config.settings['pretrained_ae'] = model_path.as_posix()

        del pretrain_dataset, aetrainer

    # forecast model train + test
    forecast_trainer = ForecastTrainer(config)

    # setup dataset
    forecast_dataset = CSP_Dataset(config)

    if config.settings['train_forecast']:
        forecast_trainer.train(forecast_dataset)

    forecast_trainer.test(forecast_dataset)

if __name__=='__main__':
    main()