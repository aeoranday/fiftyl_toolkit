from datetime import datetime
from functools import singledispatchmethod
import os

import numpy as np

import daqdataformats
from hdf5libs import HDF5RawDataFile
from rawdatautils.unpack.wibeth import np_array_adc

class Data:
    _dt_format = "%Y%m%dT%H%M%S" # datetime format from hdf5 files.
    _channels_per_link = 64
    _channel_map = np.array([112, 113, 115, 116, 118, 119, 120, 121, 123, 124, 126, 127, 64, 65, 67, 68, 70, 71, 72, 73, 75, 76, 78, 79, 48, 49, 51, 52, 54, 55, 56, 57, 59, 60, 62, 63, 0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 50, 53, 58, 61, 2, 5, 10, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 114, 117, 122, 125, 66, 69, 74, 77])

    def __init__(self, filename: str):
        self._filename = os.path.expanduser(filename)
        self._h5_file = HDF5RawDataFile(self._filename)
        self._records = self._h5_file.get_all_record_ids()
        self._last_extracted_record = None

        self.run_time = datetime.strptime(self._filename.split("_")[-1].split(".")[0], self._dt_format)
        self.run_id = int(self._filename.split('/')[-1].split('_')[1][3:])

    def get_run_id(self) -> int:
        return self.run_id

    def get_records(self) -> list:
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

            link = 0xffff & (gid >> 48)
            tmp_adc = np_array_adc(frag)

            if type(adcs) == type(None): # Now we can get the shape to initialize
                adcs = np.zeros((tmp_adc.shape[0], 128))

            for channel in range(self._channels_per_link):
                mapped_channel = np.where(self._channel_map == (link * 64 + channel))[0]
                adcs[:, mapped_channel] = tmp_adc[:, channel].reshape(-1, 1)

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
            mask = np.arange(0, 48)
        elif arg == "induction1" or arg == "induction 1" or arg == "i1" or arg == "1":
            mask = np.arange(48, 88)
        elif arg == "induction2" or arg == "induction 2" or arg == "i2" or arg == "2":
            mask = np.arange(88, 128)
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
            mask = np.array(arg, dtype='int')
        except (TypeError, ValueError):
            raise TypeError(f"{type(arg)} is not a valid array-like objecto to mask from.")
        return self._extract(self._last_extracted_record, mask)

    def __str__(self):
        return self._filename
