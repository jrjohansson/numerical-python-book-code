
# coding: utf-8

# # Chapter 1: Computing with Python

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# ## Interpreter

# In[1]:

get_ipython().run_cell_magic(u'writefile', u'hello.py', u'print("Hello from Python!")')


# In[2]:

get_ipython().system(u'python hello.py')


# In[3]:

get_ipython().system(u'python --version')


# ## Input and output caching

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

get_ipython().magic(u'pinfo math.cos')


# ## Interaction with System Shell

# In[14]:

get_ipython().system(u'touch file1.py file2.py file3.py')


# In[15]:

get_ipython().system(u'ls file*')


# In[16]:

files = get_ipython().getoutput(u'ls file*')


# In[17]:

len(files)


# In[18]:

files


# In[19]:

file = "file1.py"


# In[20]:

get_ipython().system(u'ls -l $file')


# ## Running scripts from the IPython console

# In[21]:

get_ipython().run_cell_magic(u'writefile', u'fib.py', u'\ndef fib(N): \n    """ \n    Return a list of the first N Fibonacci numbers.\n    """ \n    f0, f1 = 0, 1\n    f = [1] * N\n    for n in range(1, N):\n        f[n] = f0 + f1\n        f0, f1 = f1, f[n]\n\n    return f\n\nprint(fib(10))')


# In[22]:

get_ipython().system(u'python fib.py')


# In[23]:

get_ipython().magic(u'run fib.py')


# In[24]:

fib(6)


# ## Debugger

# In[25]:

fib(1.0)


# In[26]:

get_ipython().magic(u'debug')


# ## Timing and profiling code

# In[27]:

get_ipython().magic(u'timeit fib(100)')


# In[28]:

result = get_ipython().magic(u'time fib(100)')


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

get_ipython().magic(u'prun random_walker_max_distance(400, 10000)')


# ## IPython nbconvert

# In[36]:

get_ipython().system(u'ls ch01-code-listing.ipynb')


# In[38]:

get_ipython().system(u'ipython nbconvert --to html ch01-code-listing.ipynb')


# In[40]:

get_ipython().system(u'ipython nbconvert --to pdf ch01-code-listing.ipynb')


# In[41]:

get_ipython().run_cell_magic(u'writefile', u'custom_template.tplx', u"((*- extends 'article.tplx' -*))\n\n((* block title *)) \\title{Document title} ((* endblock title *))\n((* block author *)) \\author{Author's Name} ((* endblock author *))")


# In[42]:

get_ipython().system(u'ipython nbconvert ch01-code-listing.ipynb --to pdf --template custom_template.tplx')


# In[43]:

get_ipython().system(u'ipython nbconvert ch01-code-listing.ipynb --to python')


# # Versions

# In[22]:

get_ipython().magic(u'reload_ext version_information')
get_ipython().magic(u'version_information numpy')

