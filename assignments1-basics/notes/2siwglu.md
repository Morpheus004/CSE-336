While testing swiglu, the test suite had a case to load weights from dictionary. 
So in my implementation, I initially had `weights` as state in my Linear layer : `self.weights`

But the test cases had state loading code in the manner below
```python
layer = SwiGLU(d_model, d_ff)
layer.w1.weight.data = w1
```
Since my parameter name was weights I was getting an error. 
> Pytorch also has weight for its layers and not weights

[commit `581ffa8`](https://github.com/Morpheus004/CSE-336/commit/581ffa84309946d424d125cb326abf8b54f95aea)
