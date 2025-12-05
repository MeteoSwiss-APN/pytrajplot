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
        with open(self.file, "r") as file:
            for line in file:
                elements = line.strip().split(":", maxsplit=1)
                # Skip extraction of header information if line contains no ":"
                if len(elements) == 1:
                    continue
                key, data = elements[0], elements[1].lstrip()
                if key == "Model base time":
                    self.data["mbt"] = "".join(data)
                if key == "Model name":
                    self.data["model_name"] = "".join(data)
        # DEBUG: uncomment the following line, to check the dict
        # pprint.pprint(self.data)
