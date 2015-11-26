function Y = vl_nnreshape(X,dest_size,dzdy)
% VL_NNRESHAPE CNN reshapes input to des_size
%
%    DEST_SIZE: desired size
%
%    DZDX = VL_NNRESHAPE(X, DEST_SIZE, DZDY) computes the derivative DZDX
%    of the CNN with respect to the input X given the derivative DZDY
%    with respect to the block output Y. DZDX has the same dimension
%    as X.

% Copyright (C) 2015 Tuan-Hung VU.
% All rights reserved.
%
% This file is made available under the terms of the BSD license (see the COPYING file).

if nargin <= 2
    sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
    Y = reshape(X, [size(X,1) size(X,2) dest_size(2) size(X,3)*size(X,4)/dest_size(2)]);
else
    Y = reshape(dzdy, size(X));
end
