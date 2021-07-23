#! /bin/bash

low-priority matlab -nodisplay -sd "./" -batch "myscript(9,1)" -logfile "output_rest_9.log" &

