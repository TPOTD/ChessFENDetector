# Chess2FEN detector

# What is it?
It's a little project to try myself in CV tasks and making a small web-application.

# What does it do?
It takes your picture with 2D chessboard (eg. from your chess.com game), detects the board and convert it into FEN notation.

# How does it work?
I use YOLOv5 for chessboard detection. For FEN part I made a small neural network (I call it FENModel).
The pipeline is pretty simple: YOLO detect a chessboard, pass it to FENModel, which split the image to 64 squares (usually chessboard is 8 by 8) and then tries to predict a figure on this square. After predicting all 64 squares it concatenate all preds into one string.

# How can one try it?
You can go to http://tpotd1.eu.pythonanywhere.com/ and try it by uploading your image or checking the example.

# Does it have any problems?
My YOLO detector is pretty good in terms of finding boards, but it's result is pretty bad for the FENModel, because it simply crops either too much or too little. It's leading to FENModel to evaluate poorly, as it was trained on ideal crops. I'm trying to make a solution using Open-CV, where after detecting a board and cropping it, it's trying to find the biggest square, hopping it'll be chessboard.

