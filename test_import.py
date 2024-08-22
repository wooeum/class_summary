from image_embedding import main

test_img = r"C:\Users\USER\Pictures\Screenshots\스크린샷 2024-07-17 160254.png"
emb = main(test_img)
print(emb[0])