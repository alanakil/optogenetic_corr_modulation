#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(0,1)" -logfile "output_rest_0.log" &

