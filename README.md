# Gradient-Base-Differential-Image-Warping
式1 : ![codecogseqn-2](https://user-images.githubusercontent.com/27120804/46244585-bf73d280-c41b-11e8-8bda-3f77fb2cebf4.gif)

式2 : ![codecogseqn](https://user-images.githubusercontent.com/27120804/46244522-01504900-c41b-11e8-822d-a76611faad7e.gif)

使用したデータセット: [ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)<br>
ICL-NUIMのカメラ内部行列や各画像の世界座標系からカメラ座標系への変換行列等のMTLABコードは，すべてpythonへ書き直してMATLAB Codeディレクトリに入っている．

式1は，depthが未知であるt-1 or t+1時刻目の画像からdepthが既知であるt時刻目の画像への変換行列である．
これを用いることによってdepth未知のカラー画像をdepthが既知である画像のdepthにワーピングさせることが可能となる．

式2は変換行列の対して摂動を与えた時に勾配降下法を用いてdepth画像に一致するように計算する式である．
learning rate decay等は使っていないためlearning rateに関しては決め打ちである．（ここが結構心が折れる作業．．．）
今後, momentum等を実装していく．

# Requirement
PIL<br>
chainer<br>
