function[B, r] = generate_signal(signal, p1,p2,p3, n)

    %One brick
    if strcmp(signal, 'One_brick')
        B=zeros(p1,p2,p3);
        length=5;
        Bcore=ones(length,length,length);
        corner=[15,15,20];
        B(corner(1):corner(1)+length-1,corner(2):corner(2)+length-1,corner(3):corner(3)+length-1)=Bcore;
        r=1; %tensor rank
    elseif strcmp(signal, 'Two_bricks')
        %Two bricks
        B=zeros(p1,p2,p3);
        length=5;
        Bcore=zeros(length*2,length*2,length*2);
        Bcore(1:length,1:length,1:length)=1;
        Bcore((length+1):end,(length+1):end,(length+1):end)=1;
        corner=[15,15,20];
        B(corner(1):corner(1)+2*length-1,corner(2):corner(2)+2*length-1,corner(3):corner(3)+2*length-1)=Bcore;
        r=2; %tensor rank
    elseif strcmp(signal, '3D_Cross')
        %3D Cross
        B=zeros(p1,p2,p3);
        block=4;
        signal=1;
        Bcore=zeros(3*block,3*block,3*block);
        Bcore(1:(3*block),(block+1):(2*block),(block+1):(2*block))=signal; % 1:12,5:8,5:8
        Bcore((block+1):(2*block),1:(3*block),(block+1):(2*block))=signal; % 5:8, 1:12, 5:8
        Bcore((block+1):(2*block),(block+1):(2*block),1:(3*block))=signal; % 5:8, 5:8, 1:12
        corner=[15,15,15];
        B(corner(1):corner(1)+3*block-1,corner(2):corner(2)+3*block-1,corner(3):corner(3)+3*block-1)=Bcore;
        r=3;
    elseif strcmp(signal, 'Pyramid')
        %Pyramid
        B=zeros(p1,p2,p3);
        ht=4; %Change ht for better performance
        if n<=ht*p2
            r=ht-2;
        else
            r=ht;
        end
        Bcore=zeros(2*ht-1,2*ht-1,ht);
        for k = 1:r
            Bcore(k:(ht*2-k),k:(ht*2-k),k)=1;
        end
        corner=[15,15,20];
        B(corner(1):corner(1)+2*ht-2,corner(2):corner(2)+2*ht-2,corner(3):corner(3)+ht-1)=Bcore;
    else 
        B = zeros(p1,p2,p3);
        B(6:10,6:10,18:22) = 1;
        B(1:15,6:10,6:10) = 1;
        B(6:10,1:15,6:10) = 1;
        r = 2;
    end
end