clear all;
close all;
clc;
% pre-processing: resizing, normalizing and saving as .jpg images
F = dir('*.mat'); % selects all mat files in the working folder
q = length(F);
 for k=1:q
     a = load(F(k).name);
     im = a.cjdata.image;
     im = cat(3,im,im,im);
     sz = [299,299];             
     im = imresize(im,sz);
     s = size(im);
     imc=single(im(:));
     valMin = min(imc);
     valMax = max(imc);
     imcn=(imc-valMin)./(valMax-valMin);
     imNor=reshape(imcn, s);
     imwrite(imNor, strcat(num2str(F(k).name),'.jpg'))
  end


