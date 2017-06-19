f = 0;
id = fopen('instructions.txt','w');
for f=-12000:600:12000
fprintf(id, '%s\n', ['#Block' ]);
fprintf(id, '%s\n', ['#InstrumentAddress 20' ]);
fprintf(id, '%s\n', ['FREQ ',num2str(200000000+f+0),' Hz' ]);% fprintf(id, 'FREQ %i Hz\n',);%+30
fprintf(id, '%s\n', ['#InstrumentAddress 14' ]);
fprintf(id, '%s\n', ['FREQ ',num2str(f+1000000),' Hz' ]);
fprintf(id, '#WAIT 1000 \n');
end
fclose(id);
