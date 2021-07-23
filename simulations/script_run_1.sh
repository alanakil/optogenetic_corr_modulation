#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(1,1)" -logfile "output_rest_1.log" &

