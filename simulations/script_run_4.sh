#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(4,1)" -logfile "output_rest_4.log" &

