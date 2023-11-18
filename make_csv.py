if __name__ == "__main__":
    from data_management import make_csv
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--archive", "-a", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--include_trajectories", "-t", default=False, action="store_true")

    args = parser.parse_args()
    make_csv(
        args.output,
        archive_file=args.archive,
        include_trajectories=args.include_trajectories,
    )
