"""Command line interface of pytrajplot."""
# Standard library
import datetime
import json
import logging
import os

# Third-party
import click

# Local
from . import __version__
from .get_data import *
from .plot_altitude import alt_fig
from .utils import count_to_log_level


@click.command()
@click.argument("dir_path", type=click.Path(exists=True))
@click.option(
    "--start-prefix",
    default="startf_",
    type=str,
    help="Prefix for the start files. Default: startf_",
)
@click.option(
    "--traj-prefix",
    default="tra_geom_",
    type=str,
    help="Prefix for the start files. Default: tra_geom_",
)
@click.option(
    "--info-prefix",
    default="plot_info",
    type=str,
    help="Prefix for the plot info files. Default: plot_info",
)
def main(
    *, info_prefix: str, start_prefix: str, traj_prefix: str, dir_path: str
) -> None:
    prefix_dict = {
        "start": start_prefix,
        "trajectory": traj_prefix,
        "plot_info": info_prefix,
    }

    # plot_info_dict --> contains all the information of the plot_info file
    # trajectory_dict --> dict of dataframes, contining the comibined information of all start&trajectory files.
    # keys (list) --> contains all keys that occur in the trajectory dict (for each start/traj pair one key)
    # traj_block (dict) --> contains for each start/traj pair, number of trajectories and the number of rows that make up one individual trajectory
    # (it also exactly the same keys as the trajectory_dict, therefore the entries correspond to oneanother)

    plot_info_dict, trajectory_dict, keys, traj_block = check_input_dir(
        dir_path=dir_path, prefix_dict=prefix_dict
    )

    for key in trajectory_dict:
        print(f"key = {key}")
        trajectory_df = trajectory_dict[key]  # extract df for given key
        first_row = trajectory_df.iloc[0]  #
        number_of_times = traj_block[key]["number_of_times"]
        number_of_trajectories = traj_block[key]["number_of_trajectories"]
        x_axis = trajectory_df["time"].iloc[0:number_of_times]
        # print(f'x-axis = \n{x_axis}')
        print(
            f"key: {key}\nfirst row:\n{first_row}\nnumber_of_times: {number_of_times}\nnumber_of_trajectories: {number_of_trajectories}"
        )
        tmp = 0
        while tmp < number_of_trajectories:
            lower_row = tmp * number_of_times
            upper_row = tmp * number_of_times + number_of_times
            print(
                trajectory_df["time"][lower_row:upper_row]
            )  # check which rows are being changed
            # z_type = start_df["z_type"][tmp]
            # origin = start_df["origin"][tmp]
            # traj_df["z_type"].iloc[lower_row:upper_row] = z_type
            # traj_df["origin"].iloc[lower_row:upper_row] = origin
            tmp += 1

        # print(first_row)
        # alt_fig(key = key, trajectory = trajectory_df)
        # print(trajectory[key])
        # for origin in trajectory[key]['origin']:
        #     if origin == origin_of_interest:
        #         relevant_key = key
        #         break

    # alt_fig(plot_info=plot_info_dict, trajectory=trajectory_dict, keys=keys)

    print("--- Done.")
