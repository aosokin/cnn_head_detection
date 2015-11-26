function ind = VOChash_lookup_Casa(hash,s)

hsize=numel(hash.key);
h=mod(str2double(s([12:end])),hsize)+1;
ind=hash.val{h}(strmatch(s,hash.key{h},'exact'));
