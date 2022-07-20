import click
import typed_settings as ts
import logging

from map_prior.config import Config, Mode
import map_prior


@click.command()
@ts.click_options(Config, ts.default_loaders("map-prior", config_files=["settings.toml"]))
def main(config):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    if config.mode == Mode.TrainLightning:
        map_prior.train_lightning(config)
    if config.mode == Mode.RunFilter:
        map_prior.run_filter(config)


if __name__ == '__main__':
    main()
