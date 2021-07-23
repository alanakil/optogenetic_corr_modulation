#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(8,1)" -logfile "output_rest_8.log" &

