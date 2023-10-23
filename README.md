# FiftyL (50L) Toolkit
This package contains a useful data extraction class for use in 50L analysis. Using this package requires access to [DUNE-DAQ](https://github.com/DUNE-DAQ/daq-release/). At the moment, this package only contains the tools for __data extraction__, but it will expand when new, frequently-used tools are desired.

## Example Usage
```
import fiftyl_toolkit
data = fiftyl_toolkit.Data(path_to_data_file)
record = data.get_records()[0]

adcs = data.extract(record)
limited_adcs = data.extract(record, range(10))
plane_adcs = data.extract(record, "collection")
```
