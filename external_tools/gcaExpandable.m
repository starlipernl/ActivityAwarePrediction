function gcaExpandable

    set(gca, 'ButtonDownFcn', [...
        'set(copyobj(gca, uipanel(''Position'', [0 0 1 1])), ' ...
        '    ''Units'', ''normal'', ''OuterPosition'', [0 0 1 1], ' ...
        '    ''ButtonDownFcn'', ''delete(get(gca, ''''Parent''''))''); ']);

end