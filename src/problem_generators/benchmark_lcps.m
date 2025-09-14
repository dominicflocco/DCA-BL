function benchmark_lcps(tempdir, sizes)
%{
Generates and saves the LCP instances from the literature.

Args:
    tempdir (string): Directory to save the generated .mat files.
    sizes (vector): Array of problem sizes (n) to generate.

The function creates several types of LCP matrices (M) and vectors (q)
for each specified size, according to the problem number (pb_num).
Each generated problem is saved as a .mat file in tempdir.
%}
    pb_num = 7;
    for n = sizes
        display([num2str(n)]);
        
        e = ones(n,1);
        if pb_num == 6
            Mtilde = speye(n) + spdiags(2*ones(n,n), -n:-1, n,n);
            M = Mtilde*Mtilde.';
            q = -e;
        elseif pb_num == 7
            M = 4*speye(n) - spdiags(2*e, [1], n,n) + spdiags(e, -1, n,n) ;
            q = -e;
        elseif pb_num == 8
            M = 4*speye(n) - spdiags([e e], [-1 1], n,n);
            q = -e;
        elseif pb_num == 9
            [i, j] = find(triu(true(n), 1));
            M = speye(n) + sparse(i, j, 2, n, n); 
            q = -e;
        elseif pb_num == 10
            M = sparse(diag((1:n)/n));
            q = -e;
        elseif pb_num == 11
            M = 4*speye(n) - spdiags(2*e, [1], n,n) + spdiags(e, -1, n,n) ;
            M(1,1) = -4;
            q = e;
            q(1) = 0;
        elseif pb_num == 12
            M = 4*speye(n) - spdiags([e e], [-1 1], n,n);
            M(1,1) = -4;
            q = e;
            q(1:2) = 0;
        elseif pb_num == 13 
            [i, j] = find(triu(true(n), 1));
            M = speye(n) + sparse(i, j, 2, n, n); 
            M(n,n) = -1;
            q = -e;
            q(n) = 0;
        end
       
   
        fname = ['lcp' num2str(pb_num) '-n' num2str(n) '.mat'];
        save(fullfile(tempdir, fname),'-v7.3');
    
    end
end