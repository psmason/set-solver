# set-solver
An OpenCV solution to the set card game.  Designed to be hooked into a webcam feed. 

# What is Set?
A card game based on some simple pattern recognization rules.  As of March 2017, the New York Times hosts a daily puzzle at https://www.nytimes.com/crosswords/game/set/?page=set.  The game can be easily purchased from Amazon: http://a.co/0538gqz.  

# Output
![Sample Output](/doc/sample-output.JPG)

An image with matching sets indicated with overlayed circles.  In the sample below, red, yellow, and green circles centered on each card indicate the three sets found.  

Things aren't yet all that robust, and the top-right set was missed (see the empty squiggles).  Color recognization is a challenge in dim lighting, where the current approach is too sensitive to a purple bias in the webcam feed. 

# Debug
Passing in the "-d" flag will trigger debug mode, where the features of the centermost card will be printed to stdout.
