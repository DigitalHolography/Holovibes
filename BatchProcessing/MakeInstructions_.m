f = 0;
id = fopen('instructions.txt','w');
for f=0:20:4000
fprintf(id, 'FREQ %i Hz\n',200000000+f+30);
% fprintf(id, '%s\n', ['$FREQ ',num2str(f),' Hz' ]);
fprintf(id, '%s\n', ['$FREQ ',num2str(f),' Hz' ]);
fprintf(id, '%s\n', ['#' ]);
end
fclose(id);


