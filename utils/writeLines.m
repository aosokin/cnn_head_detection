function writeLines( fileName, lines )
%writeLines writes the cell array of strings to a text file
%
% writeLines( fileName, lines );
%
% Input: 
%   fileName - string containing the full file name
%   lines - cell array of lines contained in the file

fileID = fopen(fileName, 'w');
if fileID == -1
    error(['File ', fileName, ' can not be opened!']);
end

fprintf(fileID,'%s\n', lines{:});

fclose(fileID);

end

