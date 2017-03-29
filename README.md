# set-solver
An OpenCV solution to the set card game.  Designed to be hooked into a webcam feed. 

# Output
An image with matching sets indicated with overlayed circles.  In the sample below, red, yellow, and green circles centered on each card indicate the three sets found.  
Things aren't yet all that robus, and the top-right set was missed (see the empty squiggles).  Color recognization is a challenge in dim lighting, where the current approach biases colors to purple. 
![Sample Output](/doc/sample-output.JPG)

# Debug
Passing in the "-d" flag will trigger debug mode, where the features of the center-most card will be printed to stdout.
