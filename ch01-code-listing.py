#!/usr/bin/env python
# coding: utf-8

# # Chapter 1: Computing with Python

# Robert Johansson
# 
# Source code listings for [Numerical Python - Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib](https://link.springer.com/book/10.1007/979-8-8688-0413-7) (ISBN 979-8-8688-0412-0).

# ## Interpreter

# In[1]:


get_ipython().run_cell_magic('writefile', 'hello.py', 'print("Hello from Python!")\n')


# In[2]:


get_ipython().system('python hello.py')


# In[3]:


get_ipython().system('python --version')


# ## Input and output caching
# 
# (restart the kernel for the same output and cell numbers)

# In[1]:


3 * 3


# In[2]:


In[1]


# In[3]:


Out[1]


# In[4]:


In


# In[5]:


Out


# In[6]:


1+2


# In[7]:


1+2;


# In[8]:


x = 1


# In[9]:


x = 2; x


# ## Documentation

# In[10]:


import os


# In[11]:


# try os.w<TAB>


# In[12]:


import math


# In[13]:


get_ipython().run_line_magic('pinfo', 'math.cos')


# ## Interaction with System Shell

# In[14]:


get_ipython().system('touch file1.py file2.py file3.py')


# In[15]:


get_ipython().system('ls file*')


# In[16]:


files = get_ipython().getoutput('ls file*')


# In[17]:


len(files)


# In[18]:


files


# In[19]:


file = "file1.py"


# In[20]:


get_ipython().system('ls -l $file')


# ## Running scripts from the IPython console

# In[21]:


get_ipython().run_cell_magic('writefile', 'fib.py', '\ndef fib(N): \n    """ \n    Return a list of the first N Fibonacci numbers.\n    """ \n    f0, f1 = 0, 1\n    f = [1] * N\n    for n in range(1, N):\n        f[n] = f0 + f1\n        f0, f1 = f1, f[n]\n\n    return f\n\nprint(fib(10))\n')


# In[22]:


get_ipython().system('python fib.py')


# In[23]:


get_ipython().run_line_magic('run', 'fib.py')


# In[24]:


fib(6)


# ## Debugger

# In[25]:


fib(1.0)


# In[ ]:


get_ipython().run_line_magic('debug', '')


# ## Timing and profiling code

# In[27]:


get_ipython().run_line_magic('timeit', 'fib(100)')


# In[28]:


result = get_ipython().run_line_magic('time', 'fib(100)')


# In[29]:


len(result)


# In[30]:


import numpy as np

def random_walker_max_distance(M, N):
    """
    Simulate N random walkers taking M steps, and return the largest distance
    from the starting point achieved by any of the random walkers.
    """
    trajectories = [np.random.randn(M).cumsum() for _ in range(N)]
    return np.max(np.abs(trajectories))


# In[31]:


get_ipython().run_line_magic('prun', 'random_walker_max_distance(400, 10000)')


# ## Jupyter notebook

# In[32]:


from IPython.display import display, Image, HTML, Math


# In[33]:


Image(url='http://python.org/images/python-logo.gif')


# In[34]:


import scipy, numpy, matplotlib
modules = [numpy, matplotlib, scipy]
row = "<tr> <td>%s</td> <td>%s</td> </tr>"
rows = "\n".join([row % (module.__name__, module.__version__) for module in modules])
s = "<table> <tr><th>Library</th><th>Version</th> </tr> %s</table>" % rows


# In[35]:


s


# In[36]:


HTML(s)


# In[37]:


class HTMLDisplayer(object):
    def __init__(self, code):
        self.code = code
    
    def _repr_html_(self):
        return self.code


# In[38]:


HTMLDisplayer(s)


# In[39]:


Math(r'\hat{H} = -\frac{1}{2}\epsilon \hat{\sigma}_z-\frac{1}{2}\delta \hat{\sigma}_x')


# In[40]:


class QubitHamiltonian(object):
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta

    def _repr_latex_(self):
        return "$\hat{H} = -%.2f\hat{\sigma}_z-%.2f\hat{\sigma}_x$" % \
            (self.epsilon/2, self.delta/2)


# In[41]:


QubitHamiltonian(0.5, 0.25)


# In[42]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def f(mu):
    X = stats.norm(loc=mu, scale=np.sqrt(mu))
    N = stats.poisson(mu)
    x = np.linspace(0, X.ppf(0.999))
    n = np.arange(0, x[-1])

    fig, ax = plt.subplots()
    ax.plot(x, X.pdf(x), color='black', lw=2, label="Normal($\mu=%d, \sigma^2=%d$)" % (mu, mu))
    ax.bar(n, N.pmf(n), align='edge', label=r"Poisson($\lambda=%d$)" % mu)
    ax.set_ylim(0, X.pdf(x).max() * 1.25)
    ax.legend(loc=2, ncol=2)
    plt.close(fig)
    return fig


# In[44]:


from ipywidgets import interact
import ipywidgets as widgets


# In[45]:


interact(f, mu=widgets.FloatSlider(min=1.0, max=20.0, step=1.0));


# ## Jupyter nbconvert

# In[2]:


get_ipython().system('jupyter nbconvert --to html ch01-code-listing.ipynb')


# In[2]:


get_ipython().system('jupyter nbconvert --to pdf ch01-code-listing.ipynb')


# In[8]:


get_ipython().run_cell_magic('writefile', 'custom_template.tplx', "((*- extends 'article.tplx' -*))\n\n((* block title *)) \\title{Document title} ((* endblock title *))\n((* block author *)) \\author{Author's Name} ((* endblock author *))\n")


# In[ ]:


get_ipython().system('jupyter nbconvert ch01-code-listing.ipynb --to pdf --template custom_template.tplx')


# In[11]:


get_ipython().system('jupyter nbconvert ch01-code-listing.ipynb --to python')


# # Versions

# In[1]:


get_ipython().run_line_magic('reload_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'numpy')


# In[ ]:




