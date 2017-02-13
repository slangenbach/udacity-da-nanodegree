# Fixing MRO Makeconf setup on macOS Sierra

With default .Makeconf options I was not able to install most R packages on macOS Sierra. The following steps helped me to workaround this issue.

* Install homebrew from https://brew.sh
* Install gcc via **brew install gcc**
* Edit .Makeconf file at /Library/Frameworks/R.framework/Versions/3.3.2-MRO/Resources/etc/Makeconf
* Change **FLIBS** to FLIBS = -L/usr/local/Cellar/gcc/6.3.0_1/lib/gcc/6/gcc/x86_64-apple-darwin16.3.0 -L/usr/local/Cellar/gcc/6.3.0_1/lib/gcc/6 -lgfortran
* Change **SHLIB_LIBADD** to SHLIB_LIBADD = -L/Library/Frameworks/R.framework/Versions/3.3.2-MRO/Resources/lib
