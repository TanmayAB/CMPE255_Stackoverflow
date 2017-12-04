A python 2.7 compiler is needed to compile and run
the program.

The stated accuracy and MSLE are obtained using 2 million records.
The accuracy and MSLE might differ when using smaller training dataset.

Additionaly, the following libraries should be installed:
nltk, sklearn, numpy, pandas, scipy.

You will also need to install NLTK data if you
do not have it on your system(running nltk for the first time).
http://www.nltk.org/data.html
http://www.nltk.org/install.html

## k fold validation and plotting results
- To generate the graphs a seperate file 'Plotting_Results.py' is used.
- In the code the actual and predicted results needs to be stored in the file for all the 10 test sets 
under the name result1.dat, result2.dat respectively.
- The 'Plotting_Results.py' file will use the 10 result.dat files and plot the accuracy and Log mean square error
as a function of k. 