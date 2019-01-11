
# coding: utf-8

# In[90]:


from pypinyin import Style, pinyin


# In[91]:


path = 'data/train_all.txt'
meta = 'data/meta.csv'


# In[93]:


def ch2pinyin(txt_ch):
    ans = pinyin(txt_ch, style=Style.TONE2, errors=lambda x: x, strict=False)
    return [x[0] for x in ans if x[0] != 'EMPH_A']


# In[94]:


with open(meta, 'w') as fout:
    with open(path, 'r') as fin:
        content = fin.readlines()
        for line in content:
            parts = line[:-1].split(' ')
            wid, txt_ch = parts[0], ' '.join(ch2pinyin(parts[1:]))
            fout.write(wid + '|' + txt_ch + '|' + txt_ch + '\n')
    


# In[89]:


pinyin(['聽著嗎呢'], style=Style.TONE2, errors=lambda x: x, strict=False)

