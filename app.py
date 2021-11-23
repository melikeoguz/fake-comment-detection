import GoldenFace
Face = GoldenFace.goldenFace("tip.jpg")
color = (255,255,0)
Face.drawMask(color)
Face.writeImage("test.jpg")