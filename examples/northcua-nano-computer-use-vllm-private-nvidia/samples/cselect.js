// Replaces <select> elements with in-page dropdowns, so the open list shows up in screenshots.
document.querySelectorAll('select').forEach(sel=>{
  sel.style.display='none';
  const w=document.createElement('div');w.className='cs';w.style.cssText='position:relative;display:inline-block;width:'+(sel.dataset.w||'100%');
  const b=document.createElement('button');b.type='button';b.id=sel.id+'-btn';b.className='cs-btn';
  b.style.cssText='width:100%;height:34px;border:1px solid #c1c7d0;border-radius:5px;background:#fff;text-align:left;padding:0 30px 0 10px;font-size:14px;cursor:pointer;position:relative';
  const lab=()=>{b.innerHTML=(sel.options[sel.selectedIndex]||{}).text+'<span style="position:absolute;right:10px;top:8px;color:#666">&#9662;</span>'};lab();
  const m=document.createElement('div');m.className='cs-menu';m.id=sel.id+'-menu';
  m.style.cssText='display:none;position:absolute;left:0;right:0;top:36px;background:#fff;border:1px solid #c1c7d0;border-radius:5px;box-shadow:0 6px 18px rgba(0,0,0,.18);z-index:50;padding:4px 0';
  [...sel.options].forEach((o,i)=>{const it=document.createElement('div');it.textContent=o.text;it.className='cs-item';it.dataset.v=o.value;
    it.style.cssText='padding:7px 12px;cursor:pointer;font-size:14px';it.onmouseenter=()=>it.style.background='#e8f0fe';it.onmouseleave=()=>it.style.background='';
    it.onclick=e=>{e.stopPropagation();sel.selectedIndex=i;lab();m.style.display='none';sel.dispatchEvent(new Event('change'))};m.appendChild(it)});
  b.onclick=e=>{e.stopPropagation();document.querySelectorAll('.cs-menu').forEach(x=>{if(x!==m)x.style.display='none'});m.style.display=m.style.display==='none'?'block':'none'};
  sel.parentNode.insertBefore(w,sel);w.appendChild(b);w.appendChild(m);w.appendChild(sel);
  sel._sync=lab;
});
document.addEventListener('click',()=>document.querySelectorAll('.cs-menu').forEach(x=>x.style.display='none'));
