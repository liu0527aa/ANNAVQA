function sal_position(databasePath, height, width)
addpath('./YUVtoolbox')

resolution = [str2num(height),str2num(width)];
patchSize = 224;
frameSkip = 2;
position_width = [];
position_height = [];
for w = 1:200:resolution(1)
    if w < resolution(1) - patchSize
        for h = 1: 200: resolution(2)
            if h < resolution(2) - patchSize + 1
                position_width = [position_width, w];
                position_height = [position_height, h];
            else
                position_width = [position_width, w];
                position_height = [position_height, resolution(2) - patchSize + 1];
                break
            end
        end
    else
        for h = 1: 200: resolution(2)
            if h < resolution(2) - patchSize + 1
                position_width = [position_width, resolution(1) - patchSize + 1];
                position_height = [position_height, h];
            else
                position_width = [position_width, resolution(1) - patchSize + 1];
                position_height = [position_height, resolution(2) - patchSize + 1];
                break
            end
        end
        break
    end
end
position = int16([position_width; position_height]);

disID = fopen(databasePath);
sort_frame = [];
iFrame = 0;
while 1
    [disY, disCb, disCr] = readframeyuv420(disID, resolution(1), resolution(2));
    if feof(disID) || iFrame ==192
        break
    end 
    iFrame = iFrame + 1;
    if mod(iFrame,frameSkip)~=1
        continue
    end
    disY = reshape(disY, [resolution(2) resolution(1)])';
    disCb = reshape(disCb, [resolution(2)/2 resolution(1)/2])';
    disCr = reshape(disCr, [resolution(2)/2 resolution(1)/2])';
    disRGB = yuv2rgb(disY,disCb,disCr);

    sal_img = fes_index(disRGB);
    sal_img = imresize(sal_img,resolution(1)/size(sal_img,1),'bicubic');
    sal_sum = zeros(1,length(position));
    for iposition = 1:length(position)
        sal_sum(iposition) = sum(sum(sal_img(position(1,iposition):position(1,iposition)+patchSize-1, ...
        position(2,iposition):position(2,iposition)+patchSize-1)));
    end
    [sal_sum, sort_position] = sort(-sal_sum);

    sort_frame = [sort_frame;sort_position(1:25)];     
end

save('./test_position.mat','sort_frame');
fclose(disID);
end