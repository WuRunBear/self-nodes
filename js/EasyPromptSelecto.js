import { app } from "/scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "../../../scripts/api.js";

$el("style", {
	textContent: `
    .selfNode-model-info {
        color: white;
        font-family: sans-serif;
        max-width: 90vw;
    }
    .selfNode-model-content {
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .selfNode-model-info h2 {
        text-align: center;
        margin: 0 0 10px 0;
    }
    .selfNode-model-info p {
        margin: 5px 0;
    }
    .selfNode-model-info a {
        color: dodgerblue;
    }
    .selfNode-model-info a:hover {
        text-decoration: underline;
    }
    .selfNode-model-tags-list {
        display: flex;
        align-content: flex-start;
        flex-wrap: wrap;
        list-style: none;
        gap: 10px;
        min-height: 100px;
        height: 86%;
        overflow: auto;
        margin: 10px 0;
        padding: 0;
    }
    .selfNode-model-tags-sel-list {
        display: flex;
        align-content: flex-start;
        flex-wrap: wrap;
        list-style: none;
        gap: 10px;
        min-height: 100px;
        height: 35%;
        overflow: auto;
        margin: 10px 0;
        padding: 0;
    }
    .selfNode-model-tag {
        background-color: rgb(128, 213, 247);
        color: #000;
        display: flex;
        align-items: center;
        gap: 5px;
        border-radius: 5px;
        padding: 2px 5px;
        cursor: pointer;
    }
    .selfNode-model-tag--selected span::before {
        content: "✅";
        position: absolute;
        background-color: dodgerblue;
        top: 0;
        right: 0;
        bottom: 0;
        text-align: center;
    }
    .selfNode-model-tag:hover {
        outline: 2px solid dodgerblue;
    }
    .selfNode-model-tag p {
        margin: 0;
    }
    .selfNode-model-tag span.lable {
        text-align: center;
        border-radius: 5px;
        background-color: dodgerblue;
        color: #fff;
        padding: 2px;
        position:
         relative;
        min-width: 20px;
        overflow: hidden;
        min-width: 35px;
    }
    .selfNode-model-tag span.btn {
        text-align: center;
        border-radius: 5px;
        background-color: red;
        color: #fff;
        padding: 2px;
        position: relative;
        min-width: 20px;
        overflow: hidden;
        user-select: none;
    }
    
    .selfNode-model-metadata .comfy-modal-content {
        max-width: 100%;
    }
    .selfNode-model-metadata label {
        margin-right: 1ch;
        color: #ccc;
    }
    
    .selfNode-model-metadata span {
        color: dodgerblue;
    }
    
    .selfNode-preview {
        // max-width: 50%;
        margin-left: 10px;
        position: relative;
    }
    .selfNode-preview img {
        max-height: 300px;
    }
    .selfNode-preview .remove-button {
        position: absolute;
        font-size: 12px;
        bottom: 10px;
        right: 10px;
        color: var(--input-text);
        background-color: var(--comfy-input-bg);
        border-radius: 8px;
        border-color: var(--border-color);
        border-style: solid;
        cursor: pointer;
    }
    .selfNode-preview .reload-button {
        // position: absolute;
        font-size: 12px;
        // left: 10px;
        color: var(--input-text);
        background-color: var(--comfy-input-bg);
        border-radius: 8px;
        border-color: var(--border-color);
        border-style: solid;
        cursor: pointer;
    }
    .selfNode-model-notes {
        background-color: rgba(0, 0, 0, 0.25);
        padding: 5px;
        margin-top: 5px;
    }
    .selfNode-model-notes:empty {
        display: none;
    }    
`,
parent: document.body,
})

let pb_cache = {};
async function getPrompt(name) {
    if(pb_cache[name])
		return pb_cache[name];
	else {
        if(pb_cache[name] !== false) {
            pb_cache[name] = false;
            try {
                const resp = await api.fetchApi(`/selfNode/getPrompt?name=${name}`);
                if (resp.status === 200) {
                    let data = await resp.json();
                    pb_cache[name] = data;
                    return data;
                }
            } catch (error) {
                pb_cache[name] = undefined;
                throw error
            }
        }
        return undefined;
    }
    
}
function splitAndExtract(part) {
    if (part.startsWith('(') && part.endsWith(')')) {
        // 去掉首尾括号
        part = part.slice(1, -1);
        // 查找冒号的位置
        const colonIndex = part.indexOf(':');
        if (colonIndex!== -1) {
            // 提取冒号之前的部分
            part = part.slice(0, colonIndex);
        }
        // 替换 "("与")" 为 \
        part = part.replace(/\(/g, '\\(').replace(/\)/g, '\\)');
    }
    if (part) {
        return part;
    }
    return '';
}
// Displays input text on a node
app.registerExtension({
    name: "EasyPromptSelecto",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
		let names=['EasyPromptSelecto']
        if (names.indexOf(nodeData.name)>=0) {
            // When the node is created we want to add a readonly text widget to display the text
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);
                let prompt_type = this.widgets[this.widgets.findIndex(obj => obj.name === 'prompt_type')];
                let textEl = this.widgets[this.widgets.findIndex(obj => obj.name === 'text')];
                let category = this.widgets[this.widgets.findIndex(obj => obj.name === 'category')];

                const list = $el("ol.selfNode-model-tags-list",[]);

                let getTagList = (tags,cat) => {
                    let rlist=[]
                    Object.keys(tags).forEach((k) => {
                        if (typeof tags[k] === "string") {
                            let t=[k, tags[k]]
                            rlist.push($el(
                                "li.selfNode-model-tag",
                                {
                                    dataset: {
                                        tag: t[1],
                                        name: t[0],
                                        cat:cat,
                                        weight: 1
                                    },
                                    $: (el) => {
                                        el.onclick = () => {
                                            if (
                                                !(new RegExp(`,?\\(?${t[1]}\\)?,?`, "g")).test(textEl.value)
                                                && textEl.value.indexOf(t[1]) == -1
                                            ) {
                                                textEl.value = `${textEl.value}, ${t[1]}`;
                                            }
                                        };
                                    },
                                },
                                [
                                    $el("p", {
                                        textContent: t[0],
                                    }),
                                    $el("span.lable", {
                                        textContent: t[1],
                                    }),
                                ]
                            ))
                        }else{
                            rlist.push(...getTagList(tags[k],cat))
                        }
                    });
                    return rlist
                }

                let tags=this.addDOMWidget('tags',"list",$el('div.selfNode-preview',[
                        $el('span',[
                            $el(
                                'button.reload-button',
                                {
                                    textContent:'刷新当前列表',
                                    style:{},
                                    onclick:()=>{
                                        if(pb_cache[prompt_type.value]!=true) {
                                            pb_cache[prompt_type.value] = undefined;
                                            getPrompt(prompt_type.value)
                                                .then(data => {
                                                    if (data) {
                                                        category.options.values = Object.keys(data);
                                                        category.value = category.value? category.value : category.options.values[0];

                                                        tags.element.children[1].innerHTML='';
                                                        tags.element.children[1].append(...getTagList(data[category.value], category.value));
                                                    }
                                                });;
                                        }
                                    }
                                }
                            ),
                        ]),
                        list,
                    ]));

                const initTags = () => {
                    console.log("initTags", prompt_type.value);
                    setTimeout(() => {
                        getPrompt(prompt_type.value)
                            .then(data => {
                                if (data) {
                                    category.options.values = Object.keys(data);
        
                                    tags.element.children[1].innerHTML='';
                                    tags.element.children[1].append(...getTagList(data[category.value], category.value));
                                } else {
                                    initTags()
                                }
                            });
                    }, 500);
                }
                initTags();
                prompt_type.callback = () => {
                    console.log("prompt_type.callback", prompt_type.value);
                    getPrompt(prompt_type.value)
                        .then(data => {
                            if (data) {
                                category.options.values = Object.keys(data);
                                category.value = category.options.values[0];

                                tags.element.children[1].innerHTML='';
                                tags.element.children[1].append(...getTagList(data[category.value], category.value));
                            }
                        });
                }
                category.callback = () => {
                    console.log("category.callback", category.value);
                    if(pb_cache[prompt_type.value] && pb_cache[prompt_type.value][category.value]) {
                        tags.element.children[1].innerHTML='';
                        tags.element.children[1].append(...getTagList(pb_cache[prompt_type.value][category.value],category.value));
                    }
                }

                this.setSize([600, 700]);
                return r;
            };

        }
    },
});