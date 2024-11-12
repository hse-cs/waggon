// window.MathJax = {
//     tex2jax: {
//       inlineMath: [ ["$","$"], ["\\(","\\)"] ],
//       displayMath: [ ["$$","$$"], ["\\[","\\]"] ]
//     },
//     TeX: {
//       TagSide: "right",
//       TagIndent: ".8em",
//       MultLineWidth: "85%",
//       equationNumbers: {
//         autoNumber: "AMS",
//       },
//       unicode: {
//         fonts: "STIXGeneral,'Arial Unicode MS'"
//       }
//     },
//     displayAlign: "center",
//     showProcessingMessages: false,
//     messageStyle: "none"
//   };

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})