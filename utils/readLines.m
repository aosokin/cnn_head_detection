function lines = readLines( fileName )
%readLines reads the file line by line
%
% lines = readLines( fileName );
%
% Input: 
%   fileName - string containing the full file name
%
% Output:
%   lines - cell array of lines contained in the file

fileID = fopen(fileName, 'r');
if fileID == -1
    error(['File ', fileName, ' can not be opened!']);
end
lines = textscan(fileID, '%s\n');
fclose(fileID);

lines = lines{1};

end

