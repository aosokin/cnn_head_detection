function ind = VOChash_lookup_HH(hash,s)

hsize=numel(hash.key);
h=mod(str2double(s([5:7 9:end])),hsize)+1;
ind=hash.val{h}(strmatch(s,hash.key{h},'exact'));
