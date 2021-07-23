#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(5,1)" -logfile "output_rest_5.log" &

