
initialImage = im2single( imread('peppers.png') );

width = size(initialImage, 2);
height = size(initialImage, 1);

cropPositions = [ 1, 1, height, width; ... % full image
                  1, 1, height / 2, width / 3; ... % sub patch
                  -height / 2, -width / 2, height / 2, width / 3 ]; ... % sub patch not inside the image
crops = cropRectanglesMex( initialImage, cropPositions, [height, width] );

figure(1), imshow( initialImage );
figure(2), imshow( crops(:,:,:,1) );
figure(3), imshow( crops(:,:,:,2) );
figure(4), imshow( crops(:,:,:,3) );

