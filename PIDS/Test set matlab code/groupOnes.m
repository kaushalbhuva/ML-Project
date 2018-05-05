function grpNo = groupOnes(filtdiffXPM)

grpNo = 0;
for i = 1:length(filtdiffXPM)-1
    if filtdiffXPM(i)>filtdiffXPM(i+1)
        grpNo = grpNo + 1;
    end
end

if grpNo == 0
    grpNo = 1; % to avoid division by zero; Anyway this means slope did not cross threshold
end