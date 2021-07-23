#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(6,1)" -logfile "output_rest_6.log" &

