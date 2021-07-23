#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(3,1)" -logfile "output_rest_3.log" &

