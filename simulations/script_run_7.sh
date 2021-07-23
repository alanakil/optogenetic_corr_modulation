#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(7,1)" -logfile "output_rest_7.log" &

