#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(2,1)" -logfile "output_rest_2.log" &

