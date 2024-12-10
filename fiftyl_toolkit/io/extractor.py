from ..ChannelMaps import CHANNEL_MAPS, PLANE_MAPS

from datetime import datetime
from functools import singledispatchmethod
import os

import numpy as np
from numpy.typing import NDArray

import daqdataformats
from hdf5libs import HDF5RawDataFile
from rawdatautils.unpack.wibeth import np_array_adc

class Data:
    _channels_per_link = 64

    def __init__(self, filename: str, map_name: str="2T-UX"):
        self._filename = os.path.expanduser(filename)
        self._h5_file = HDF5RawDataFile(self._filename)
        self._records = self._h5_file.get_all_record_ids()
        self._last_extracted_record = None

        self._creation_timestamp = int(self._h5_file.get_attribute("creation_timestamp"))
        self._run_id = self._h5_file.get_int_attribute("run_number")
        self._file_index = self._h5_file.get_int_attribute("file_index")
        self.set_channel_map(map_name)
        self.set_plane_map(map_name)
        return

    def get_channel_map(self) -> list[int]:
        return self._channel_map

    def set_channel_map(self, map_name: str="2T-UX") -> None:
        try:
            self._channel_map = np.array(CHANNEL_MAPS[map_name])
            self._inverse_map = np.argsort(self._channel_map)
        except KeyError:
            raise KeyError(f"Given channel map name is not available. Use one of: {list(CHANNEL_MAPS.keys())}")
        return

    def get_plane_map(self) -> NDArray:
        return self._plane_map

    def set_plane_map(self, map_name: str="2T-UX") -> None:
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
        """
        Return the run ID integer.
        """
        return self._run_id

    @property
    def file_index(self):
        return self._file_index

    @file_index.getter
    def file_index(self) -> int:
        """
        Return the sub-run ID integer.
        """
        return self._file_index

    @property
    def records(self):
        return self._records

    @records.getter
    def records(self) -> list[tuple[int, int]]:
        """
        Return the list of records contained in this file.
        """
        return self._records

    def extract(self, record, *args) -> np.ndarray:
        """
        Extract data from the initialized HDF5 data file.
            record
                Trigger record to extract from the current dataset.
            args
                Channels to extract from, array-like, plane names, channel number, or empty for all.
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

    def _extract(self, record, mask):
        """
        Performs the extraction after all the preprocessing.
        """
        geo_ids = self._h5_file.get_geo_ids(record)
        adcs = None # Don't know the shape of the upcoming fragment, so prepare later

        for gid in geo_ids:
            frag = self._h5_file.get_frag(record, gid)

            link = (0xffff & (gid >> 48)) % 2
            map_bounds = (link * 64, (link+1) * 64)
            tmp_adc = np_array_adc(frag)

            if adcs is None: # Now we can get the shape to initialize
                adcs = np.zeros((tmp_adc.shape[0], 128))
            elif tmp_adc.shape[0] < adcs.shape[0]:  # New fragment is smaller than the old. Make old smaller.
                adcs = adcs[:tmp_adc.shape[0], :]
            elif tmp_adc.shape[0] > adcs.shape[0]:  # New fragment is larger than the old. Make new smaller.
                tmp_adc = tmp_adc[:adcs.shape[0], :]

            adcs[:, self._inverse_map[map_bounds[0]:map_bounds[1]]] = tmp_adc

        return adcs[:, mask]

    @singledispatchmethod
    def _extract_helper(self, arg: type(None)):
        """
        Get all channels.
        """
        mask = np.arange(0, 128)
        return self._extract(self._last_extracted_record, mask)

    @_extract_helper.register
    def _(self, arg: int):
        """
        Get only one channel.
        """
        mask = arg
        return self._extract(self._last_extracted_record, mask)

    @_extract_helper.register
    def _(self, arg: str):
        """
        Get by plane name.
        """
        arg = arg.lower()
        if arg == "collection" or arg == "collect" or arg == "c":
            mask = np.array(self._plane_map["collection"])
        elif arg == "induction1" or arg == "induction 1" or arg == "i1" or arg == "1":
            mask = np.array(self._plane_map["induction1"])
        elif arg == "induction2" or arg == "induction 2" or arg == "i2" or arg == "2":
            mask = np.array(self._plane_map["induction2"])
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
        ## Multiple planes by name case
        if len(arg) <= 3: # Check if strings were given, such as ('collection', 'i2')
            strings = [type(s) == str for s in arg]
            if np.all(strings):
                adcs = None
                for plane in arg:
                    if type(adcs) == type(None):
                        adcs = self._extract_helper(plane)
                    else:
                        adcs = np.hstack((adcs, self._extract_helper(plane)))
                return adcs

        ## Integer array-like masking
        try:
            mask = np.array(arg, dtype=int)
        except (TypeError, ValueError):
            raise TypeError(f"{type(arg)} is not a valid array-like objecto to mask from.")
        return self._extract(self._last_extracted_record, mask)

    def __str__(self):
        return self._filename
