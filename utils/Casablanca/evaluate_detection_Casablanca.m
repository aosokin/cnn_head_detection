function [rec, prec, ap] = evaluate_detection_Casablanca(det, id, VOCopts, classname)
fprintf('VOC evaluation...');
opath = sprintf(VOCopts.detrespath, id, classname);
fid = fopen(opath, 'w');
for i=1:length(det)
    f_name = det(i).id;
    for j=1:size(det(i).bb,1)
        fprintf(fid, '%s %f %f %f %f %f\n', f_name, det(i).bb(j, 5),...
            det(i).bb(j, 1), det(i).bb(j, 2),...
            det(i).bb(j, 1)+det(i).bb(j, 3)-1, det(i).bb(j, 2)+det(i).bb(j, 4)-1);
    end
end
fclose(fid);

[rec, prec, ap]=VOCevaldet_Casablanca(VOCopts, id, classname, false);

end