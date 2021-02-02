from os.path import realpath, join


class folder:

    @staticmethod
    def get_working_dir(keyword: str = "hotel_cloud") -> str:

        paths = []
        for i in realpath(__file__).split("/"):
            if i:  # avoid empty str
                if i != keyword:
                    paths.append(i)
                else:
                    paths.append(i)
                    break

        return join("/", *paths)
