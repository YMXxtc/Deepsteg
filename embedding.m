function embedding(cost_dir)
% Read the input cover image
cover_dir = '/data/ymx/ymx/BOSSBase/BossBase-1.01-cover-resample-256';
%cost_dir = '/data/ymx/ymx/mine/Res/Res_KB3_1divLPFmean3HFcover8LFmean11Mul';  




tStart = tic;
for Payload = 0.4:0.4
    %stego_dir = '/data/ymx/ymx/mine/Res/Res_SRM30_1divLF3HFcoverLF3MMulAddmeanM';
    stego_dir = cost_dir;
    stego_dir = sprintf('%s_%.1fbpp',stego_dir,Payload);

    if ~exist(stego_dir,'dir')
        mkdir(stego_dir)
        stego_dir
    end
    
    changeRate = 0;

    for index = 1:10000
        cover_path = sprintf('%s/%d.pgm', cover_dir, index);
        Cover = double(imread(cover_path));
        %size(Cover)

        Cost = load(sprintf('%s/%d.mat', cost_dir, index));
        Cost = Cost.cost;
        Cost = squeeze(Cost);
        %Cost = reshape(Cost,[1,256,256]);
        %size(Cost);
    %Payload = 0.3;

            

             % embedding
            [Stego, prob_map, rhoP1, rhoM1] = EMB(Cover, Payload, Cost); 

            temp = sum(sum(Stego - Cover ~= 0)) / numel(Cover)
            changeRate = changeRate + temp;

            stego_path = sprintf('%s/%d.pgm', stego_dir, index);
            imwrite(uint8(Stego), stego_path);

            modify_path = sprintf('%s/%d_modify.pgm', stego_dir, index);
            imwrite(uint8((Stego-Cover+1)*127), modify_path);

            modify_png_path = sprintf('%s/%d_modify.png', stego_dir, index);
            imwrite(uint8((Stego-Cover+1)*127), modify_png_path);

            p_path = sprintf('%s/%d_probas.pgm', stego_dir, index);
            imwrite(uint8(prob_map*255), p_path);



    end
    tEnd = toc(tStart);

    changeRate = changeRate / 10000.0
    fprintf('Embedding is done in: %f (sec)\n',tEnd);

    file_id = fopen('embedding_log.txt','a');
    fprintf(file_id,'%s %s-%.1f: %.4f\n', datestr(now),cost_dir, Payload, changeRate);
    fclose(file_id);

end
end


function [stego, prob_map, rhoP1, rhoM1] = EMB(cover, payload, cost)

wetCost = 10^10;

% compute embedding costs \rho
rhoB = cost;
rhoB(rhoB > wetCost) = wetCost; % threshold on the costs
rhoB(isnan(rhoB)) = wetCost; % if all xi{} are zero threshold the cost

rhoP1 = rhoB;
rhoM1 = rhoB;
rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
[stego, prob_map] = EmbeddingSimulator(cover, rhoP1, rhoM1, payload*numel(cover), false);


%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound).
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
function [y, prob_map] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)

    n = numel(x);
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    if fixEmbeddingChanges == 1
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
    else
        RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    end
    randChange = rand(size(x));
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;

    %y = uint8(y);
    prob_map = pChangeP1 + pChangeM1;


    function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;

        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            m3 = ternary_entropyf(pP1, pM1); % m3=0
           

            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                
                return;
            end
        end


        l1 = 0;
        m1 = double(n); % m1=n=65536
        lambda = 0;

        alpha = double(message_length)/n; % alpha=0.4
        
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
            
            lambda = l1+(l3-l1)/2;
            pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            m2 = ternary_entropyf(pP1, pM1);
            

            if m2 < message_length
                l3 = lambda;
                m3 = m2;
            else
                l1 = lambda;
                m1 = m2;
            end
            iterations = iterations + 1;
            %disp(lambda)
        end

        %disp(lambda) % lambda=30.3802
    end




    function Ht = ternary_entropyf(pP1, pM1)
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end
end
end
