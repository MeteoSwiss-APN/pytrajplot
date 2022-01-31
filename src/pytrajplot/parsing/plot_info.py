"""plot info file support."""

# Standard library
import pprint
from typing import Any
from typing import Dict


class PLOT_INFO:
    """Support plot_info files.

    Attributes:
        data: Data part of the atab file.

    """

    def __init__(self, file) -> None:
        """Create an instance of ``PLOT_INFO``.

        Args:
            file: Input file.

            sep (optional): Separator for data.

        """
        # Set instance variables
        self.file = file
        self.data: Dict[str, Any] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse the plot info file."""
        # read the plot_info file
        with open(self.file, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        # remove first element, corresponds to empty line
        del lines[0]

        # iterate through the file, extract relevant information into dict
        while len(lines) > 0:
            line = lines.pop(0)
            elements = line.strip().split(":", maxsplit=1)
            # Stop extraction of header information if line contains no ":"
            if len(elements) == 1:
                break
            key, data = elements[0], elements[1].lstrip()
            if key == "Model base time":
                self.data["mbt"] = "".join(data)
            if key == "Model name":
                self.data["model_name"] = "".join(data)
        # DEBUG: uncomment the following line, to check the dict
        # pprint.pprint(self.data)
