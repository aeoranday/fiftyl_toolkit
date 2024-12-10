from ..ChannelMaps import CHANNEL_MAPS, PLANE_MAPS

from functools import singledispatchmethod
import os
from types import NoneType

import numpy as np
from numpy.typing import NDArray, ArrayLike

from daqdataformats import Fragment
from hdf5libs import HDF5RawDataFile
from rawdatautils.unpack.wibeth import np_array_adc


class Data:
    """
    Data reader for experiments that have happened at CERN B182 2-003.

    Given a data file to process, this class is able to read the data using various queries, such as by plane name,
    channel number, and all channels.
    """
    _channels_per_link = 64

    def __init__(self, filename: str, map_name: str = "2T-UX"):
        """
        Load the given data file in :filename: and set preliminary attributes.

        Parameters:
            filename (str) : Path of data file to process.
            map_name (str) : Channel map to use when extracting. Defaults to latest map available.
        """
        self._filename: str = os.path.expanduser(filename)
        self._h5_file: HDF5RawDataFile = HDF5RawDataFile(self._filename)
        self._records: list[tuple[int, int]] = self._h5_file.get_all_record_ids()
        self._last_extracted_record: tuple[int, int] | None = None

        self._creation_timestamp: int = int(self._h5_file.get_attribute("creation_timestamp"))
        self._run_id: int = self._h5_file.get_int_attribute("run_number")
        self._file_index: int = self._h5_file.get_int_attribute("file_index")
        self._channel_map: NDArray | None = None
        self._plane_map: dict[str, range] | None = None

        self.set_channel_map(map_name)
        self.set_plane_map(map_name)
        return

    def get_channel_map(self) -> NDArray[np.int_]:
        if self._channel_map is None:
            raise ValueError("Channel map was not set yet.")
        return self._channel_map

    def set_channel_map(self, map_name: str = "2T-UX") -> None:
        """
        Set the channel map to use when extracting data.

        Parameters:
            map_name (str) : Name of the map to use. fiftyl_toolkit.ChannelMaps has the available maps.
        """
        try:
            self._channel_map = np.array(CHANNEL_MAPS[map_name])
            self._inverse_map: NDArray = np.argsort(self._channel_map)
        except KeyError:
            raise KeyError(f"Given channel map name is not available. Use one of: {list(CHANNEL_MAPS.keys())}")
        return

    def get_plane_map(self) -> dict[str, range]:
        if self._plane_map is None:
            raise ValueError("Plane map was not set yet.")
        return self._plane_map

    def set_plane_map(self, map_name: str = "2T-UX") -> None:
        """
        Set the plane map to use when extract.

        Map naming is consistent with the channel map variant.

        Parameters:
            map_name (str) : Name of the map to use. fiftyl_toolkit.ChannelMaps has the available maps.
        """
        try:
            self._plane_map = PLANE_MAPS[map_name]
        except KeyError:
            raise KeyError(f"Given plane map name is not available. Use one of: {list(PLANE_MAPS.keys())}")
        return

    @property
    def run_id(self):
        return self._run_id

    @run_id.getter
    def run_id(self) -> int:
        """ The run number for the given data file. """
        return self._run_id

    @property
    def file_index(self):
        return self._file_index

    @file_index.getter
    def file_index(self) -> int:
        """ The file index of the given data file. """
        return self._file_index

    @property
    def creation_timestamp(self):
        return self._creation_timestamp

    @creation_timestamp.getter
    def creation_timestamp(self) -> int:
        """ Epoch timestamp for when the file was created. """
        return self._creation_timestamp

    @property
    def records(self):
        return self._records

    @records.getter
    def records(self) -> list[tuple[int, int]]:
        """ A list of records within the given file. """
        return self._records

    def extract(self, record: tuple[int, int], *args) -> NDArray:
        """
        Extract data from the given HDF5 data file.

        Parameters:
            record (tuple[int, int]) : Trigger record to extract from the current dataset.
            *args (int | ArrayLike | str | None) : Channels to extract from, array-like, plane names, channel number, or empty for all.

        Returns a 2D np.ndarray of channel waveforms.
        """
        if record in self._records:
            self._last_extracted_record = record
        else:
            raise IndexError("This record ID is not available in the current data set.")

        if len(args) == 0:
            arg = None
        else:
            arg = args[0]
        return self._extract_helper(arg)

    def _extract(self, record: tuple[int, int], mask: NDArray | int) -> NDArray:
        """
        Performs the extraction after all the preprocessing.
        """
        geo_ids: set[int] = self._h5_file.get_geo_ids(record)
        adcs: NDArray | None = None  # Don't know the shape of the upcoming fragment, so prepare later
        if len(geo_ids) == 0:
            raise ValueError("No links to process.")

        for gid in geo_ids:
            frag: Fragment = self._h5_file.get_frag(record, gid)

            link: int = (0xffff & (gid >> 48)) % 2
            map_bounds: tuple[int, int] = (link * 64, (link+1) * 64)
            tmp_adc: NDArray = np_array_adc(frag)

            if adcs is None:  # Now we can get the shape to initialize
                adcs = np.zeros((tmp_adc.shape[0], 128))
            elif tmp_adc.shape[0] < adcs.shape[0]:  # New fragment is smaller than the old. Make old smaller.
                adcs = adcs[:tmp_adc.shape[0], :]
            elif tmp_adc.shape[0] > adcs.shape[0]:  # New fragment is larger than the old. Make new smaller.
                tmp_adc = tmp_adc[:adcs.shape[0], :]

            adcs[:, self._inverse_map[map_bounds[0]:map_bounds[1]]] = tmp_adc

        assert isinstance(adcs, np.ndarray)
        return adcs[:, mask]

    @singledispatchmethod
    def _extract_helper(self, arg: None):
        """
        Get all channels.
        """
        mask = np.arange(0, 128)
        assert isinstance(self._last_extracted_record, tuple)
        return self._extract(self._last_extracted_record, mask)

    @_extract_helper.register
    def _(self, arg: int):
        """
        Get only one channel.
        """
        mask = arg
        assert isinstance(self._last_extracted_record, tuple)
        return self._extract(self._last_extracted_record, mask)

    @_extract_helper.register
    def _(self, arg: str):
        """
        Get by plane name.
        """
        if self._plane_map is None:
            raise ValueError("Plane map was not yet set.")

        arg = arg.lower()
        if arg == "collection" or arg == "collect" or arg == "c":
            mask = np.array(self._plane_map["collection"])
        elif arg == "induction1" or arg == "induction 1" or arg == "i1" or arg == "1":
            mask = np.array(self._plane_map["induction1"])
        elif arg == "induction2" or arg == "induction 2" or arg == "i2" or arg == "2":
            mask = np.array(self._plane_map["induction2"])
        assert isinstance(self._last_extracted_record, tuple)
        return self._extract(self._last_extracted_record, mask)

    # Union typing supported in Python >=3.11, so this will have to do for now.
    @_extract_helper.register(set)
    @_extract_helper.register(list)
    @_extract_helper.register(tuple)
    @_extract_helper.register(range)
    @_extract_helper.register(np.ndarray)
    def _(self, arg):
        """
        Get by valid array-like object.
        """
        # Multiple planes by name case
        if len(arg) <= 3:  # Check if strings were given, such as ('collection', 'i2')
            strings = [isinstance(s, str) for s in arg]
            if np.all(strings):
                adcs = None
                for plane in arg:
                    if adcs is None:
                        adcs = self._extract_helper(plane)
                    else:
                        adcs = np.hstack((adcs, self._extract_helper(plane)))
                return adcs

        # Integer array-like masking
        try:
            mask = np.array(arg, dtype=int)
        except (TypeError, ValueError):
            raise TypeError(f"{type(arg)} is not a valid array-like objecto to mask from.")
        return self._extract(self._last_extracted_record, mask)

    def __str__(self):
        """ Data file path given to process. """
        return self._filename

    def __len__(self):
        """ Length of the records for the given data file. """
        return len(self._records)
